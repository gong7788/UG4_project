(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b3)
		(clear b3)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b1 t0)
		(in-tower b1)
		(on b0 b1)
		(in-tower b0)
		(on b2 b0)
		(in-tower b2)
		(on b4 b2)
		(in-tower b4)
		(on b5 b4)
		(in-tower b5)
		(blue b3)
		(red b5)
		(blue b5)
		(blue b6)
		(blue b7)
		(blue b8)
		(blue b9)
		(red b3)
		(red b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b6 b5)) (not (on b3 b5)))))
)