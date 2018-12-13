(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(clear b9)
		(in-tower t0)
		(on b3 t0)
		(in-tower b3)
		(on b5 b3)
		(in-tower b5)
		(on b2 b5)
		(in-tower b2)
		(on b4 b2)
		(in-tower b4)
		(on b9 b4)
		(in-tower b9)
		(red b0)
		(red b1)
		(red b2)
		(blue b9)
		(blue b4)
		(blue b3)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (and (not (on b1 b4)) (not (on b0 b4)) (not (on b8 b9)))))
)