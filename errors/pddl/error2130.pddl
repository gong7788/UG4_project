(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(clear b2)
		(on-table b3)
		(clear b3)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(on b4 b6)
		(in-tower b4)
		(on b5 b4)
		(in-tower b5)
		(on b2 b5)
		(in-tower b2)
		(blue b0)
		(blue b1)
		(red b3)
		(blue b4)
		(red b5)
		(red b6)
		(red b7)
		(blue b9)
		(blue b5)
		(red b2)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (and (not (on b5 b6)) (not (on b3 b5)) (not (on b3 b2)))))
)