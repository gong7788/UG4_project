(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(clear b3)
		(on-table b8)
		(clear b8)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b7 b9)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(on b5 b6)
		(in-tower b5)
		(on b4 b5)
		(in-tower b4)
		(on b3 b4)
		(in-tower b3)
		(red b0)
		(blue b2)
		(red b3)
		(blue b3)
		(red b6)
		(blue b6)
		(red b7)
		(blue b7)
		(red b8)
		(blue b9)
		(red b1)
		(blue b4)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (and (not (on b8 b9)) (not (on b8 b7)) (not (on b8 b6)) (not (on b8 b5)) (not (on b8 b4)) (not (on b8 b3)))))
)